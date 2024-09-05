import { useForm } from '@mantine/form';
import {
  TextInput,
  PasswordInput,
  Text,
  Paper,
  Group,
  Button,
  Divider,
  Checkbox,
  Anchor,
  Stack,
} from '@mantine/core';
import { useMutation } from '@tanstack/react-query';
import { GoogleButton } from '../../components/Buttons/GoogleButton';
import { GithubButton } from '../../components/Buttons/GithubButton';
import classes from './Signup.module.css'
import { signUp } from '../../api/auth';
import { useNavigate } from 'react-router-dom';

export default function Signup(props) {
  const navigate = useNavigate();
  const form = useForm({
    initialValues: {
      email: '',
      first_name: '',
      last_name: '',
      password: '',
      terms: true,
      username: ''
    },

    validate: {
      email: (val) => (/^\S+@\S+$/.test(val) ? null : 'Invalid email'),
      password: (val) => (val.length <= 6 ? 'Password should include at least 6 characters' : null),
    },
  });

  const mutation = useMutation({
    mutationFn: signUp,
    onSuccess: () => {
      navigate('/');
    }
  })



  const handleSubmit = async(e) => {
    e.preventDefault();
    mutation.mutate(form.getValues())
  }
  
  return (
    <div className={classes.container}>
      <Paper radius="md" p="xl" withBorder {...props}>
      <Text size="lg" fw={500}>
        Welcome to SensorGrid, register with
      </Text>

      <Group grow mb="md" mt="md">
        <GoogleButton radius="xl">Google</GoogleButton>
        <GithubButton radius="xl">Github</GithubButton>
      </Group>

      <Divider label="Or continue with email" labelPosition="center" my="lg" />

      <form onSubmit={handleSubmit}>
        <Stack>
          <Group
            grow
           >
          <TextInput
              label="First Name"
              placeholder="First name"
              value={form.values.first_name}
              onChange={(event) => form.setFieldValue('first_name', event.currentTarget.value)}
              radius="md"
              required
            />
            <TextInput
              label="Last Name"
              placeholder="Last name"
              value={form.values.last_name}
              onChange={(event) => form.setFieldValue('last_name', event.currentTarget.value)}
              radius="md"
              required
            />
          </Group>
          
          <TextInput
              label="Username"
              placeholder="Enter username"
              value={form.values.username}
              onChange={(event) => form.setFieldValue('username', event.currentTarget.value)}
              radius="md"
              required
            />

          <TextInput
            required
            label="Email"
            placeholder="hello@mantine.dev"
            value={form.values.email}
            onChange={(event) => form.setFieldValue('email', event.currentTarget.value)}
            error={form.errors.email && 'Invalid email'}
            radius="md"
          />

          <PasswordInput
            required
            label="Password"
            placeholder="Your password"
            value={form.values.password}
            onChange={(event) => form.setFieldValue('password', event.currentTarget.value)}
            error={form.errors.password && 'Password should include at least 6 characters'}
            radius="md"
          />

            <Checkbox
              label="I accept terms and conditions"
              checked={form.values.terms}
              onChange={(event) => form.setFieldValue('terms', event.currentTarget.checked)}
            />
        </Stack>
        {
          mutation.isError && (
            <Text  c='#a90003' pt='sm' style={{textAlign: 'center'}}>{mutation.error.message}</Text>
          )
        }
        <Group justify="space-between" mt="xl">
          <Anchor component="" type="button" c="dimmed"  size="xs" >
            {
               'Already have an account? Login'
            }
          </Anchor>
          <Button type="submit" radius="xl">
            Submit
          </Button>
        </Group>
      </form>
    </Paper>
    </div>
  );
}