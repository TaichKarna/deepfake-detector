import useStore from "../store/store"
export default function Home(){
    const user = useStore( (state) => state.user);
    console.log(user);
    
    return (
       <>
        <p>Home</p>
        </>
    )
}